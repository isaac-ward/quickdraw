"""Shared builders: cfg -> dataclasses, model, datasets, loaders. Used by all entrypoints."""

from __future__ import annotations

import re
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


def step_fps(cfg, ecfg) -> float:
    """The TRUE sample rate of anything the model rolls out, in Hz -- use this for every video of a
    prediction, never `1/ecfg.dt`.

    `ecfg.dt` is the dataset's FRAME period (`_recorded_dt` reads it from summary.json), because that
    is what velocity and physics quantities scale with. But ONE autoregressive step spans
    `data.subsample` frames, so a rollout's frames are `subsample/dt` apart, and encoding them at
    `1/dt` plays the video back `subsample`x too fast with nothing on screen to give it away.
    Derived from subsample rather than configured, for the same reason `dt_eff` is (see below): a
    hand-set rate silently goes stale the moment the stride changes."""
    return (1.0 / float(ecfg.dt)) / max(1, int(cfg.data.get("subsample", 1) or 1))


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
                              dt=_recorded_dt(cfg, float(e.dt)),
                              position_idx=(list(e.position_idx) if e.get("position_idx", None) is not None else None),
                              velocity_idx=(list(e.velocity_idx) if e.get("velocity_idx", None) is not None else None),
                              dynamics_prior=bool(e.get("dynamics_prior", False)),
                              quat_idx=(list(e.quat_idx) if e.get("quat_idx", None) is not None else None),
                              mass=float(e.get("mass", 12000.0)), raw_dt=float(e.get("raw_dt", 0.05)),
                              dt_eff=float(e.get("dt_eff", 0.25)))
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


# Model-SIZE presets (mm_flow): model.size -> the hidden capacity knobs it sets. NOT tuned for every problem — a
# starting point. d/heads = backbone width + attention heads (head_dim=d/heads must be a POWER OF 2 and >=16 for the
# compiled FlexAttention -> small uses heads=12 so 192/12=16); num_tokens = image latent tokens; decode_base = U-Net
# decoder width. Keys in _SIZE_MODEL_KEYS live at model level; the rest are set on the image modality.
# COMPATIBILITY WITH THE PRETRAINED DEFAULT (2026-08-11): mm_flow now defaults to the frozen TAESD, whose latent
# is 4*16*16 = 1024 floats, and the adapter wants num_tokens*d == 1024 EXACTLY.
#   mini   8*128 = 1024  -> EXACT. The only preset that fits.
#   tiny   8*64  =  512  -> RAISES (LOSSY: the bag cannot hold the latent). Needs modalities.<img>.pretrained=false.
#   small 16*192 = 3072  -> builds, but PADDED: 67% of the bag carries no latent. HOW BAD depends on the norm:
#                           under layernorm padding is ACTIVE damage (-4.4 dB measured) because _ln is per TOKEN,
#                           so the zeros enter the mean/std the real floats are divided by; under affine there is
#                           no bag LN at all, so zero-pad + strip is an exact bijection and the cost is only
#                           wasted width -- the denoiser still spends capacity on dims that get stripped. Either
#                           way prefer num_tokens*d == 1024.
SIZE_PRESETS = {
    "tiny":  {"d": 64, "heads": 4, "num_tokens": 8, "decode_base": 16},      # d=64/heads=4 -> head_dim 16 (min power-of-2); narrow U-Net decoder
    "mini":  {"d": 128, "num_tokens": 8, "decode_base": 16},                 # heads=8 -> head_dim 16; narrow (16) U-Net decoder, wider backbone than tiny
    "small": {"d": 192, "heads": 12, "num_tokens": 16, "decode_base": 64},   # heads=12 -> head_dim 16 (power of 2)
}
_SIZE_MODEL_KEYS = ("d", "heads", "depth")
_SIZE_BASE = {"d": 128, "heads": 8, "depth": 4, "num_tokens": 8, "decode_base": 32}  # base defaults, to detect
#   clashes. MUST TRACK conf/model/mm_flow.yaml: d moved 32 -> 128 on 2026-08-11 (the default trunk became the
#   frozen TAESD, whose 1024-float latent forces num_tokens*d == 1024). If this disagrees with the yaml, every
#   `model.size=...` raises a bogus "you also overrode model.d=..." clash.


def apply_size_preset(cfg):
    """model.size=tiny|small -> set the preset's hidden capacity knobs (model.d/heads + the image modality's
    num_tokens/decode_base) IN PLACE on cfg, so config.resolved records the real values. RAISES if you ALSO overrode
    one of those knobs individually (size vs explicit-knob clash -> pick one). Image knobs apply only when an image
    modality is present. No-op if model.size is unset. Call ONCE, early (before the resolved dump), NOT on resume."""
    from omegaconf import open_dict
    m = cfg.get("model", None)
    size = None if m is None else m.get("size", None)
    if not size:
        return
    if size not in SIZE_PRESETS:
        raise ValueError(f"model.size={size!r} is unknown; options: {sorted(SIZE_PRESETS)}")
    preset = SIZE_PRESETS[size]
    img = next((md for md in (m.get("modalities") or []) if md.get("kind") == "image"), None)
    clashes = []
    for k, v in preset.items():
        holder = m if k in _SIZE_MODEL_KEYS else img
        if holder is None:                                    # image key but no image modality -> skip
            continue
        cur = holder.get(k, _SIZE_BASE.get(k))
        if cur not in (_SIZE_BASE.get(k), v):
            clashes.append(f"model.{'' if k in _SIZE_MODEL_KEYS else 'modalities.<image>.'}{k}={cur}")
    if clashes:
        raise ValueError(f"model.size={size} sets {sorted(preset)}, but you also overrode {clashes}. "
                         f"Use model.size OR the individual knob(s), not both — remove one.")
    for k, v in preset.items():
        if k in _SIZE_MODEL_KEYS:
            m[k] = v
        elif img is not None:
            with open_dict(img):
                img[k] = v


def _proprio_prior_mode(cfg):
    """The proprio modality's decode-PRIOR mode (none|identity|physics), the env-supplied selector for the unified
    decode/physics path. Reads model.modalities.<proprio>.prior; falls back to 'physics' when the LEGACY
    environments.dynamics_prior=true is set with no explicit prior field (backward compat)."""
    e = cfg.get("environments", {}) or {}
    legacy = bool(e.get("dynamics_prior", False) if hasattr(e, "get") else False)
    for m in (cfg.model.get("modalities", None) or []):
        mm = (lambda k, v: m.get(k, v)) if hasattr(m, "get") else (lambda k, v: getattr(m, k, v))
        if str(mm("name", "")) == "proprio":
            p = str(mm("prior", "none") or "none")
            return p if p != "none" else ("physics" if legacy else "none")
    return "physics" if legacy else "none"


def _make_dynamics_prior(cfg):
    """The decode-prior callable f(prev_obs, action) -> next_obs for the proprio modality, or None. The ENV owns
    the prior: 'physics' -> owm's a=R(q)F/m (environments/owm_physics.py); 'identity' -> copy prev (learned-delta
    baseline); 'none' -> None (byte-identical). Generalizes the old environments.dynamics_prior gate."""
    mode = _proprio_prior_mode(cfg)
    if mode == "none":
        return None
    if mode == "identity":
        return lambda prev, act: prev.float()          # obs_next = prev + head(token); env supplies no physics
    e = cfg.get("environments", {}) or {}
    from ..environments.owm_physics import dynamics_prior as _dp
    pos = list(e.position_idx)
    vel = list(e.velocity_idx) if e.get("velocity_idx", None) is not None else [i + len(pos) for i in pos]
    quat = list(e.quat_idx)
    # Body rate for the exact attitude kinematics q'=q(x)exp(1/2 w dt). ego-13: quat [6,7,8,9] -> rate [10,11,12].
    # Explicit override via environments.bodyrate_idx; else the three dims after quat.
    bodyrate = (list(e.bodyrate_idx) if e.get("bodyrate_idx", None) is not None
                else [quat[-1] + 1 + i for i in range(3)])
    # Rotational dynamics w'=w+I^-1(tau*raw_dt-(wxIw)*dt_eff): torque=action[3:6], inertia from env physics.
    # Both default-off (None) -> quat integrates the copied rate only (attitude kinematics), bodyrate copied.
    torque = (list(e.torque_idx) if e.get("torque_idx", None) is not None else [3, 4, 5])
    inertia = (list(e.inertia_diag) if e.get("inertia_diag", None) is not None else [80000.0, 80000.0, 50000.0])
    # Body-rate clamp bounding the quadratic gyroscopic term in the AR rollout (see owm_physics.RATE_CLAMP).
    # Config-settable via environments.rate_clamp; default 1.0 rad/s (~30x the ~0.03 data range -> non-binding).
    rate_clamp = float(e.get("rate_clamp", 1.0))
    mass, raw_dt = float(e.get("mass", 12000.0)), float(e.get("raw_dt", 0.05))
    # DERIVE dt_eff from subsample (audit finding #1): action is SUMMED over the subsample window so Δv uses
    # raw_dt, but position integrates over the FULL subsampled-step duration = subsample*raw_dt. Hardcoding 0.25
    # was only right for subsample=5; derive it so the physics can't silently drift if subsample changes.
    sub = int(cfg.data.get("subsample", 1) or 1)
    dt_eff = sub * raw_dt
    return lambda prev, act: _dp(prev, act, pos, vel, quat, bodyrate_idx=bodyrate,
                                 torque_idx=torque, inertia_diag=inertia, rate_clamp=rate_clamp,
                                 mass=mass, raw_dt=raw_dt, dt_eff=dt_eff)


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
            if head_dim < 16 or (head_dim & (head_dim - 1)) != 0:   # compiled FlexAttention: power of 2 AND >= 16
                raise ValueError(
                    f"model.compile_rollout needs head_dim (d/heads) to be a POWER OF 2 and >= 16 for the compiled "
                    f"FlexAttention kernel (head_dim=24 fails to compile), but d={m.d}/heads={m.heads} -> "
                    f"head_dim={head_dim}. Pick d/heads giving head_dim in {{16,32,64}} (e.g. d=192/heads=12 -> 16), "
                    f"or disable compile_rollout (eager has no such constraint).")
        common = dict(specs=specs, d=m.d, depth=m.depth, heads=m.heads, window=m.window,
                      mlp_ratio=m.mlp_ratio, rope_theta=m.rope_theta, action_dim=effective_action_dim(cfg),
                      grad_checkpoint=bool(m.get("grad_checkpoint", False)),
                      compile_rollout=compile_rollout,
                      latent_norm=m.get("latent_norm", "layernorm"),
                      action_fourier_freqs=int(m.get("action_fourier_freqs", 0)),
                      action_squash=str(m.get("action_squash", "none")),
                      # relative-position encoding (floor lever): OFF by default. position_idx are the proprio
                      # dims that are world position (from the env); relative_scale is their normalized
                      # within-window displacement std (the rescale-to-unit-variance gain denominator).
                      relative_position=bool(m.get("relative_position", False)),
                      position_idx=(cfg.environments.get("position_idx", None) if m.get("relative_position", False) else None),
                      relative_scale=(list(m.get("relative_scale")) if m.get("relative_scale", None) is not None else None),
                      dynamics_prior=_make_dynamics_prior(cfg))
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
        # dynamics_follows_p_tf: meaningful only where the dynamics loss CONDITIONS ON a context it can swap.
        # RAISE rather than silently ignore -- a user who sets this believes they changed the training regime
        # (same rule as the fail-fast block just below). See design/flow.md.
        # p_tf_dynamics: probability the DYNAMICS loss conditions on the truth. Accepts the LEGACY boolean
        # `dynamics_follows_p_tf` so configs written before 2026-08-25 (and checkpoints adopted by
        # run_standalone) rebuild what they trained under: false == 1.0 (always clean), true == None (follow
        # p_tf). Translated loudly rather than silently, and the new key wins if both are present.
        _p_tf_dyn = m.get("p_tf_dynamics", 1.0)
        _legacy = m.get("dynamics_follows_p_tf", None)
        if _legacy is not None and "p_tf_dynamics" not in m:
            _p_tf_dyn = None if bool(_legacy) else 1.0
            print(f"[config] legacy model.dynamics_follows_p_tf={_legacy} -> p_tf_dynamics="
                  f"{_p_tf_dyn!r} (false==1.0 always-clean, true==None follow-p_tf)", flush=True)
        if _p_tf_dyn is not None:
            _p_tf_dyn = float(_p_tf_dyn)
            if not 0.0 <= _p_tf_dyn <= 1.0:
                raise ValueError(f"model.p_tf_dynamics must be in [0,1] or null; got {_p_tf_dyn}")
        _dfp = None if (_p_tf_dyn is not None and _p_tf_dyn >= 1.0) else True
        if _dfp is not None:
            if name in ("mm_dsar", "dsar", "base"):
                raise ValueError("model.p_tf_dynamics has no meaning for mm_dsar: it has no dynamics loss at "
                                 "all (loss_terms returns {}). Remove the key.")
            if name in ("mm_lsar", "lsar"):
                raise ValueError("model.p_tf_dynamics is not implementable for mm_lsar: its dynamics loss "
                                 "scores the ROLLOUT'S OUTPUT against the encoded true future, so there is no "
                                 "context to pin to clean latents -- pinning it would compare encode(fut) "
                                 "against itself (identically zero). mm_lsar already follows p_tf by "
                                 "construction; remove the key.")
        # Fail fast on config knobs that only one model reads (otherwise silently ignored).
        if cfg.get("collapse", None) is not None and name not in ("mm_lsar", "lsar"):
            raise ValueError(f"model.collapse=... (a collapse-prevention strategy) is only used by the LSAR model "
                             f"(model.name in mm_lsar/lsar); got {name!r}. DSAR is grounded by its data-space "
                             f"re-encode and Flow by its reconstruction, so neither takes a collapse strategy. "
                             f"Remove the collapse override or switch to mm_lsar.")
        _ah = m.get("action_head", {}) or {}
        _ah_on = bool(_ah.get("enabled", False) if hasattr(_ah, "get") else getattr(_ah, "enabled", False))
        if _ah_on and name not in ("mm_flow", "flow"):
            raise ValueError(f"model.action_head.enabled=true (the action-flow MPPI prior) is only wired on the "
                             f"Flow model (model.name in mm_flow/flow); got {name!r}. TODO: we intend to port the "
                             f"action-flow head to the other prediction mechanisms; until then, set "
                             f"action_head.enabled=false here.")
        # probabilistic prediction heads (parametric distribution over the next latent; design/models/
        # probabilistic_heads.md). dist_head only means something for the mm_dist model -> fail fast elsewhere.
        if m.get("dist_head", None) is not None and name != "mm_dist":
            raise ValueError(f"model.dist_head=... (a parametric distribution head) is only used by the "
                             f"probabilistic model (model.name=mm_dist); got {name!r}. Set model.name=mm_dist "
                             f"or remove the dist_head override.")
        if name == "mm_dist":
            from ..models.multimodal import MultiModalDistribution
            from ..models.dist_heads import make_dist_head
            # ONE prediction mechanism: the parametric head is mutually exclusive with the flow/diffusion head.
            if m.get("diffusion", None) is not None:
                raise ValueError("model.name=mm_dist with a model.diffusion block: pick ONE prediction mechanism "
                                 "(the parametric distribution head OR the rectified-flow/diffusion head). Remove "
                                 "model.diffusion, or switch to model.name=mm_flow.")
            cv = (cfg.get("variations") or {}).get("contraction", {}) or {}
            cw = float((cv.get("weight", 0.0) if hasattr(cv, "get") else getattr(cv, "weight", 0.0)) or 0.0)
            if cw > 0.0:   # the sample/argmax step has no double-backward Jacobian (same reason as the flow guard)
                raise ValueError("variations.contraction is mutually exclusive with model.name=mm_dist: the "
                                 "sampling/argmax step has no double-backward Jacobian for the power-iteration. "
                                 "Disable contraction (weight=0).")
            return MultiModalDistribution(**common, head=make_dist_head(m),
                                          stochastic_eval=bool(m.get("stochastic_eval", True)))
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
                                       stochastic_eval=bool(dfg("stochastic_eval", True)),   # matches the
                                       #   class default + conf/model/mm_flow.yaml: train and eval roll on the
                                       #   same distribution. A config MISSING the key (e.g. an older run's
                                       #   saved config adopted by run_standalone) must not silently differ.
                                       time_sampling=str(dfg("time_sampling", "uniform")),
                                       flow_hidden=int(dfg("flow_hidden", 0)),
                                       flow_arch=str(dfg("flow_arch", "mlp")),
                                       flow_arch_depth=int(dfg("flow_arch_depth", 2)),
                                       flow_arch_heads=int(dfg("flow_arch_heads", 4)),
                                       # FALLBACK False on purpose: a config that does not MENTION this key is
                                       # an OLD config (e.g. adopted from a checkpoint's logs/config.json by
                                       # run_standalone), and it was trained without the extra channel -- so
                                       # rebuilding it with True doubles the flow's h_dim and the checkpoint
                                       # fails to load with a size mismatch. New runs get True from mm_flow.yaml.
                                       concat_action_embedding=bool(dfg("concat_action_embedding", False)),
                                       lambda_flow=m.get("lambda_flow", 1.0),
                                       lambda_consistency=m.get("lambda_consistency", 1.0),
                                       df_scale=df_scale, df_granularity=df_granularity,
                                       action_head_enabled=bool(ahg("enabled", False)),
                                       action_head_weight=float(ahg("weight", 1.0)),
                                       action_head_shortcut=bool(ahg("shortcut", True)),
                                       action_head_detach_gradient=bool(ahg("detach_gradient", False)),
                                       action_head_chunk=int(ahg("chunk", 1)),
                                       dynamics_detach_encoder=bool(m.get("dynamics_detach_encoder", False)),
                                       # default 1.0 (always-clean) so an old config adopted by
                                       # run_standalone rebuilds the behaviour it TRAINED under.
                                       p_tf_dynamics=_p_tf_dyn)
        raise ValueError(f"unknown model.name: {name!r}")


_hf_root_cache: dict = {}   # hf_repo -> snapshot path (avoid re-resolving/downloading per call)


def autobatch_find(cfg, device, log=print) -> int:
    """Size `data.batch` to the largest whose AR training step fits `VRAM - autobatch_reserve_gb`, capped at
    `autobatch_max`. Sizes by a LINEAR FIT of peak-vs-batch (2 probes + confirm), falling back to a
    bracket-and-bisect search; budgets on RESERVED memory (what OOMs), not allocated. The AR step is DISPATCH-bound (accelerations.md Exp 8), so bigger batch is nearly-free
    throughput until memory binds — pick the largest that fits. Builds a THROWAWAY model +
    SYNTHETIC batches (memory is shape- not value-dependent), probes real `rollout_train` fwd + decode + bwd,
    then frees. Probes the REAL training step (rollout_train -> recon + flow loss, backward, AND an AdamW
    step so optimizer states count) — the earlier synthetic pow(2) proxy under-counted the loss graph +
    optimizer by ~12 GB and picked batches that OOM'd.

    The eval phase is PROBED and REPORTED (probe_eval) -- reported, NOT gated: it only warns, because eval
    routines self-disable after two failures and refusing to start a run on an ESTIMATE is the worse trade.
    Sizing therefore uses the TRAINING constraint alone, which is correct and measured: eval memory does not
    depend on data.batch, the phases are sequential, and the freed training blocks are reused (2026-08-19: eval
    after an 87.5GB train peak added NO new reservation). The margin held back is `data.autobatch_reserve_gb`
    PLUS the GPU-resident dataset, which is subtracted explicitly below.
    Config: data.autobatch{,_reserve_gb,_max,_base}."""
    import gc
    # ONE ABSOLUTE MARGIN (2026-08-18), replacing `autobatch_headroom` (a fraction). The two reserves that used
    # to exist covered the same thing, and after this change neither has its original job: fragmentation is now
    # MEASURED (we budget on reserved, not allocated) and the eval spike is now GATED (probe_eval below). What is
    # left is one residual -- allocator growth over a full epoch beyond a 2-iteration probe, evidenced at +12 GB
    # by the "batch 112 probed 81GB then OOM'd at 93GB" incident. That is a property of the ALLOCATOR, not of the
    # card, so it must be absolute: a 12 GB spike is 12 GB whether the card is 40 GB or 100 GB, whereas a 25%
    # fraction is 10 GB on one and 25 GB on the other. `autobatch_headroom` is deliberately DELETED from
    # conf/data so stale `data.autobatch_headroom=...` overrides fail LOUDLY instead of silently doing nothing.
    reserve = float(cfg.data.get("autobatch_reserve_gb", 12.0)) * 1e9
    cap = int(cfg.data.get("autobatch_max", 512))
    base = int(cfg.data.get("autobatch_base", 16))
    total = torch.cuda.get_device_properties(device).total_memory

    def _resident_frame_bytes():
        """The image frame store MMWindowLoader parks on the GPU -- allocated AFTER this function has already
        spent the card (train_world_model: autobatch_find at ~line 150, window_loaders at ~172).

        This was the entire unexplained "probe vs real" gap that `autobatch_reserve_gb` was standing in for.
        MEASURED 2026-08-19 on the 128px/4Hz config: probe 87.29GB allocated + 3.170GB resident = 90.46 against
        a real ep1 peak of 90.29GB reserved -- residual ~0.17GB, i.e. there is no allocator drift to reserve
        against at all. It is deterministic and computable, and it scales with dataset-hours x img_size**2, so a
        FIXED reserve cannot cover it: at 20h/256px it is tens of GB while the knob stays put. Frames only (the
        dominant term, 2.87 of 3.17GB); the ~0.3GB of window/index tensors stays inside the blind reserve."""
        try:
            from ..data.dataset import get_subsample, load_split_episodes
            # This estimate rides on the PROCESS-GLOBAL frame stride, because the loader applies it. If the
            # caller has not called set_subsample() yet, the lengths come back unsubsampled and the estimate is
            # off by exactly that factor -- measured: 14.18GB instead of 2.83GB at subsample=5, i.e. the
            # verification harness silently shrank the batch. Over-estimating is the SAFE direction, but say so.
            _want, _have = int(cfg.data.get("subsample", 1) or 1), get_subsample()
            if _have != _want:
                log(f"[autobatch] WARNING: data.subsample={_want} but the process frame stride is {_have} "
                    f"(set_subsample() not called yet) -- the resident frame-store estimate below is {_want/_have:.0f}x "
                    f"too LARGE, so the chosen batch will be conservative. Call set_subsample() before sizing.")
            imgs = [sp for sp in specs if getattr(sp, "kind", "vector") == "image"]
            if not imgs:
                return 0
            n = 0
            for split in ("train", "val"):
                eps_ = load_split_episodes(resolve_data_root(cfg), split,
                                           repo_id=cfg.data.get("repo_id", "torus"))
                n += sum(len(o) for o, _ in eps_)          # subsampling is applied inside the loader
            b = 0
            for sp in imgs:
                sz = getattr(sp, "img_size", 128)
                hw = (sz, sz) if isinstance(sz, int) else tuple(int(x) for x in sz)
                b += n * hw[0] * hw[1] * int(getattr(sp, "channels", 3))   # uint8
            return b
        except Exception as e:
            log(f"[autobatch] could not size the resident frame store ({type(e).__name__}: {e}); "
                f"falling back to the blind reserve alone")
            return 0
    # No floor on the budget (the `max(total*0.25, ...)` guard was deleted 2026-08-20): it only bound when the
    # reserve exceeded 75% of the card, and when it bound it silently GRANTED more than the operator asked to
    # hold back -- inverting the knob. A nonsensical reserve now makes every probe fail and the RuntimeError
    # below fires, and that message already names autobatch_reserve_gb as the thing to lower.
    P, F = int(cfg.data.P), int(cfg.data.F); L = P + F
    de = int(cfg.model.get("detach_every", 16)); rf = float(cfg.model.get("recon_frac", 1.0))
    specs = _modality_specs(cfg); adim = effective_action_dim(cfg)
    # AFTER specs: _resident_frame_bytes closes over it (defining the budget earlier raised NameError -- caught
    # by running the verification, which reported "could not size the resident frame store" and silently fell
    # back to the blind reserve, exactly the kind of quiet degradation this whole pass is about).
    resident = _resident_frame_bytes()
    budget = int(total - reserve - resident)
    log(f"[autobatch] budget {budget/1e9:.1f}GB = card {total/1e9:.1f} - reserve {reserve/1e9:.1f} - "
        f"resident frame store {resident/1e9:.2f} | allocator={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
    model = build_model(cfg).to(device).train()
    # The eager binary search sizes MEMORY, which is ~compile-independent (validation: 22.9 GB compiled vs eager),
    # so probe EAGER — else the compiled path recompiles at every probe batch size (~96 s each = thrash). If
    # compile_rollout is on, confirm_compiled() re-enables it and validates the chosen batch once on the real
    # compiled step (stepping down on OOM), so the returned batch is checked on the exact path training will run.
    compiled_run = bool(cfg.model.get("compile_rollout", False))
    model.compile_rollout = False

    # real optimizer so its Adam states (fp32 moments) count toward the probed peak — they allocate on the
    # first .step(). lr=1e-12 so repeated probe steps don't drift the throwaway weights to NaN (which would
    # break the value-sensitive loss path); states allocate regardless of lr.
    opt = torch.optim.AdamW(model.parameters(), lr=1e-12, weight_decay=0.0, fused=True)

    def synth(B):
        obs = {}
        for s in specs:
            if getattr(s, "kind", "vector") == "image":
                sz = getattr(s, "img_size", 128)
                hw = (sz, sz) if isinstance(sz, int) else tuple(int(x) for x in sz)
                obs[s.name] = torch.rand(B, L, hw[0], hw[1], 3, device=device)
            else:
                obs[s.name] = torch.randn(B, L, int(getattr(s, "dim", 6)), device=device)
        act = torch.randn(B, L, adim, device=device)   # act_seq is length L (P+F): _par_preds does act[:, :-1] -> L-1
        return obs, act                                 #   (aligns with obs[:, :-1]); _seq_preds does act[:, :L-1]

    def _loss_from_preds(preds_and_feeds, obs, act):
        # (preds, feeds) pair, NEVER a side-channel: probe() runs _seq_preds AND _par_preds, and a stashed
        # `feeds` from the first would be backwarded through a graph the first backward already freed
        # ("Trying to backward through the graph a second time"). Caught in smoke, 2026-08-25.
        preds, feeds = preds_and_feeds
        # recon_losses (recon_frac subset, all-head decode) + loss_terms (flow) — the memory-relevant loss graph,
        # SHARED by the sequential (p_tf=0 rollout) and parallel (p_tf=1) probes. Mirrors LitWorldModel._step.
        future = {k: v[:, P:] for k, v in obs.items()}
        recon_src = preds if getattr(model, "pred_obs_in_loss", True) else preds.detach()
        if rf < 1.0:
            Tf = recon_src.shape[1]; k = max(1, int(round(rf * Tf)))
            idx = torch.randperm(Tf, device=device)[:k]
            src, futr = recon_src[:, idx], {kk: v[:, idx] for kk, v in future.items()}
        else:
            src, futr = recon_src, future
        recon, rw = model.recon_losses(src, futr)
        raw, w = model.loss_terms(preds, future, obs, 0.0, act, **({"feeds": feeds} if feeds is not None else {}))
        return sum(w[k] * raw[k] for k in raw) + sum(rw[k] * recon[k] for k in recon)

    def _seq_preds(obs, act):   # p_tf=0 AUTOREGRESSIVE rollout (the in-rollout epochs, i.e. most of training)
        # Uses the SHARED-ENCODE fast path, because that is what LitWorldModel._step actually runs at p_tf==0
        # (lit.py: `share = (p_tf == 0.0) and hasattr(m, "flow") and all noise_std == 0` -> encode once, pass
        # z_full[:, :P] as precomputed_ctx). Probing WITHOUT it modelled a step the trainer never executes and
        # over-estimated: measured 2026-08-19, probe/real = 1.007 on bsp32mse but 1.32 on anch128 -- i.e. on a
        # frozen-AE config it threw away a THIRD of the card. Guarded exactly as lit.py guards it.
        share = hasattr(model, "flow") and all(model.modalities[k].noise_std == 0 for k in obs)
        z_full = model.encode_state(obs) if share else None
        # return_feeds when the model's dynamics loss consumes them, so the probe keeps MIRRORING _step:
        # with dynamics_follows_p_tf on, the real step's loss graph includes the feeds-conditioned context.
        # Probe-vs-real drift is exactly how this finder previously picked batches that OOM'd.
        wf = bool(getattr(model, "dynamics_follows_p_tf", False))
        out = model.rollout_train({k: v[:, :P] for k, v in obs.items()}, act[:, : L - 1],
                                  {k: v[:, P:] for k, v in obs.items()}, 0.0, de,
                                  precomputed_ctx=(z_full[:, :P] if share else None), return_feeds=wf)
        return out if wf else (out, None)     # ALWAYS a (preds, feeds) pair -- see _loss_from_preds

    def _par_preds(obs, act):   # p_tf=1 PARALLEL forward (epoch-0 regime: whole sequence in ONE pass — often the
        #                         TRUE memory peak). feeds=None: no rollout ran, so there is nothing to condition on.
        return model({k: v[:, :-1] for k, v in obs.items()}, act[:, :-1])[:, P - 1:], None

    def _probe_path(obs, act, preds_fn):   # 2 iters (so Adam states allocate) of fwd-loss + bwd + step -> peak bytes
        torch.cuda.synchronize(device); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        for _ in range(2):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                try:
                    loss = _loss_from_preds(preds_fn(obs, act), obs, act)   # faithful full loss graph
                except Exception as e:                                     # value-sensitive path choked on synth data
                    if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower():
                        raise
                    loss = preds_fn(obs, act)[0].float().pow(2).mean()     # fall back to a pred-only estimate
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        # RESERVED, not allocated (2026-08-18). The allocator's reserved pool is what actually OOMs -- the gap is
        # fragmentation, and budgeting on `allocated` made that gap invisible and left it to be absorbed by a
        # hand-tuned headroom fraction. On record: "batch 112 probed 81GB then OOM'd at 93GB". Both are returned
        # so the ratio is logged rather than assumed.
        return torch.cuda.max_memory_reserved(device), torch.cuda.max_memory_allocated(device)

    def probe(B):   # peak = MAX(p_tf=0 rollout, p_tf=1 parallel forward). The old probe measured ONLY the rollout,
        try:        #   missing the parallel-forward peak that epoch 0 hits -> under-sized -> OOM at epoch 0. None on OOM.
            obs, act = synth(B)
            a, b_ = _probe_path(obs, act, _seq_preds), _probe_path(obs, act, _par_preds)
            return max(a[0], b_[0]), max(a[1], b_[1])       # (reserved, allocated), each maxed over both paths
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if not (isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()):
                raise
            model.zero_grad(set_to_none=True); opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache(); return None

    def fits(B):
        pr = probe(B)
        if pr is None:
            return False, None
        res, alloc = pr
        if alloc > 0:
            log(f"[autobatch] probe b={B}: reserved {res/1e9:.1f}GB / allocated {alloc/1e9:.1f}GB "
                f"(frag {100*(res-alloc)/max(alloc,1):.0f}%) vs budget {budget/1e9:.0f}GB")
        return res <= budget, res

    def done(b):   # free probe activations before the caller builds loaders + the real model
        gc.collect(); torch.cuda.empty_cache()
        # RESET the peak counter (2026-08-18): the probes legitimately allocate far more than training will ever
        # use (a rejected batch-16 probe hit 82.2 GB on one config), and nothing downstream reset it -- so every
        # mem/* metric for the whole run was really "max since process start, including probes we threw away".
        torch.cuda.reset_peak_memory_stats(device)
        return b

    def confirm_compiled(b):   # validate the eager-chosen batch on the REAL compiled step (recompiles per shape)
        if not compiled_run:
            return b
        model.compile_rollout = True
        last_ok = None                 # largest batch measured as FITTING on the compiled step
        # FIXED 2026-08-18. Two bugs: (1) it stepped down by a CONSTANT 8, which on the batch<=16 configs this
        # repo actually runs is a >=50% jump; (2) when `b - 8 < 1` it logged "over budget down to batch {b};
        # using {b}" and RETURNED b -- a batch it had just measured as NOT fitting. At b=8 that was the branch
        # taken, i.e. every bespoke run. Now the step is PROPORTIONAL to the measured overshoot and an
        # over-budget batch is never returned.
        for _ in range(6):
            log(f"[autobatch] compiled-confirm: probing batch {b} on the compiled step (one-time ~1-2 min compile)...")
            ok_, p_ = fits(b)
            if ok_:
                last_ok = b
                log(f"[autobatch] compiled-confirm OK: batch {b} ({(p_ or 0)/1e9:.1f}/{budget/1e9:.0f}GB compiled)")
                return b
            if b <= 1:
                # The EAGER probe accepted a larger batch, so a compiled failure at b=1 is a compile-time
                # transient rather than a real capacity result. Warn loudly and proceed at 1 rather than raise,
                # so this cannot turn a previously-working config into a hard startup failure.
                log("[autobatch] WARNING: compiled step over budget even at batch 1 — the eager probe accepted "
                    "more, so this is likely a compile-time transient. Using 1; raise data.autobatch_reserve_gb "
                    "or set data.autobatch=false data.batch=<n> if this run OOMs.")
                return 1
            nb = max(1, int(b * budget / p_)) if p_ else b // 2      # step PROPORTIONAL to the overshoot
            nb = min(nb, b - 1)                                       # always make progress
            log(f"[autobatch] compiled step over budget at {b} "
                f"({(p_ or 0)/1e9:.1f}/{budget/1e9:.0f}GB) — stepping down to {nb}")
            b = nb
        # Return the last batch that actually FIT, never the last one PROBED (2026-08-20). This path used to
        # return `b` -- a batch just measured as over budget -- contradicting this function's own claim that
        # "an over-budget batch is never returned". Reachable whenever 6 proportional step-downs do not converge.
        if last_ok is not None:
            log(f"[autobatch] compiled-confirm exhausted its retries; using the last batch that FIT ({last_ok})")
            return last_ok
        log("[autobatch] compiled-confirm exhausted its retries and NOTHING fit; falling back to batch 1")
        return 1

    # ---- Step 4 (2026-08-18): PROBE THE INFERENCE PHASE ---------------------------------------------------
    # Until now eval memory was never measured -- it was covered by a headroom fraction, and the documented
    # remedy for an eval OOM was "raise the fraction". Worse, the number people cited as the eval cost
    # (epoch_peak - probe_peak, ~+4.5 GB) was NOT a measurement of eval at all: nothing reset the CUDA peak
    # counter, so the epoch peak was a process max that also included autobatch's own rejected probes.
    #
    # KEY PROPERTY that makes this a REPORT rather than a subtraction: eval memory does NOT depend on data.batch.
    # It is fixed by the eval config (ood_horizon hardcodes n_ep=8 at routines.py:133, plus eval.horizon,
    # ae_floor_episodes, closed_loop_steps) and by the model. Training activations are freed before eval and the
    # allocator reuses those blocks, so the two phases do not add -- each must independently fit the budget.
    def _synth_eval(B, T_ctx, H):
        obs = {}
        for sp in specs:
            if getattr(sp, "kind", "vector") == "image":
                sz = getattr(sp, "img_size", 128)
                hw = (sz, sz) if isinstance(sz, int) else tuple(int(x) for x in sz)
                obs[sp.name] = torch.rand(B, T_ctx, hw[0], hw[1], 3, device=device)
            else:
                obs[sp.name] = torch.randn(B, T_ctx, int(getattr(sp, "dim", 6)), device=device)
        return obs, torch.randn(B, T_ctx + H, adim, device=device)

    def probe_eval():
        """Peak bytes of the eval phase, by running the REAL entry point (`imagine_eval`) at the REAL shapes
        (`ood_horizon_shapes`). Returns (reserved, allocated) or None / -1.0 on OOM.

        REWRITTEN 2026-08-20. The previous version hand-rolled the rollout and diverged from the routine in five
        ways -- no autocast, no decode_chunk, use_cache=False, a hardcoded n_ep=8, and a 256 horizon cap -- which
        made it over-model open_loop by 2.4x (39.5GB vs the real 16.4GB) while not modelling closed_loop_16 at
        all, which is where the real peak actually is. Its apparent agreement with a real run was coincidence.
        Now it calls the same function the routine calls, per mode, and takes the max."""
        try:
            from ..data.dataset import load_split_episodes
            from ..evaluation.routines import ood_horizon_shapes
            from .setup import resolve_data_root                                    # noqa: F401 (same module)
            img_heads = [sp.name for sp in specs if getattr(sp, "kind", "vector") == "image"]
            # episode LENGTHS only -- the proprio loader reads no frames, so this is cheap (seconds)
            ep_lens = [len(o) for o, _ in load_split_episodes(resolve_data_root(cfg), "val",
                                                             repo_id=cfg.data.get("repo_id", "torus"))]
            n_ep, H, _cl_h, modes, calls = ood_horizon_shapes(cfg, bool(img_heads), ep_lens, P)
            dc = int(cfg.eval.get("decode_chunk", 64) or 0) or None
            heads = [sp.name for sp in specs]
            worst, worst_name = (0, 0), "-"
            for name, rows, hz in calls:
                torch.cuda.synchronize(device); torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                obs, act = _synth_eval(rows, P, hz)
                with torch.no_grad():
                    model.imagine_eval(obs, act, hz, heads=heads, decode_chunk=dc)
                r = (torch.cuda.max_memory_reserved(device), torch.cuda.max_memory_allocated(device))
                log(f"[autobatch] eval probe {name}: rows={rows} horizon={hz} -> "
                    f"{r[0]/1e9:.1f}GB reserved / {r[1]/1e9:.1f}GB allocated")
                del obs, act
                if r[0] > worst[0]:
                    worst, worst_name = r, name
            log(f"[autobatch] eval probe WORST mode = {worst_name} at {worst[0]/1e9:.1f}GB reserved "
                f"(n_ep={n_ep}, H={H}, decode_chunk={dc})")
            return worst[0]
        except Exception as e:
            oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()
            log(f"[autobatch] eval probe {'OOM' if oom else 'skipped'} ({type(e).__name__}: {str(e)[:140]})")
            return -1.0 if oom else None
        finally:
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)

    _ev = probe_eval()
    if _ev == -1.0:
        log("[autobatch] WARNING: the eval phase OOM'd on its own, INDEPENDENT of data.batch. Training will "
            "still be sized (correctly -- eval cost does not depend on the batch), but expect eval routines to "
            "fail and self-disable. Shrink eval.horizon / eval.closed_loop_steps / eval.ae_floor_episodes, or "
            "turn off manifold.")
    elif _ev is not None and _ev > budget:
        # WARN, never raise: eval routines self-disable after two failures, so refusing to start a run on an
        # ESTIMATE is the worse trade. Sizing ignores this number by design -- eval memory is batch-independent
        # and the phases are sequential with block reuse (measured: eval after an 87.5GB train peak added no new
        # reservation), so a smaller batch could not rescue an eval that genuinely does not fit.
        log(f"[autobatch] WARNING: the eval phase ({_ev/1e9:.1f}GB) exceeds the {budget/1e9:.0f}GB budget on its "
            f"own. Training is sized independently and will still run; eval routines may fail and self-disable.")

    # ---- Step 3 (2026-08-18): LINEAR-FIT SIZING, with the bisection below as fallback ---------------------
    # peak(B) = a*B + b. `b` is persistent (fp32 weights + grads + Adam m,v = 16 B/param); `a` is per-sample
    # activations. Measured on bsp32mse: 41.2GB @ B=8 and 82.2GB @ B=16 -> a=5.125 GB/sample, b=0.20GB, i.e.
    # essentially pure-linear-through-the-origin because activations dominate a 6.4M-param model entirely.
    #
    # WHY THIS RATHER THAN BISECTION, and the reason is SAFETY not speed: bisection's base-fits branch walks a
    # doubling ladder (32, 64, ... autobatch_max=512) and EVERY RUNG IS A REAL ALLOCATION -- batch 16 already
    # reserved 82.2 of a 95.8GB card on one config, and a proprio-only config climbs to the cap. On a shared box
    # that threatens the CO-RESIDENT run. A fit probes SMALL and jumps straight to the answer, never allocating
    # far above what it will choose. It also yields a reusable GB/sample model instead of one search result.
    def _persistent_bytes():
        n = sum(q.numel() for q in model.parameters())
        return n * 16          # fp32 param + grad + Adam m + Adam v

    def _fit_predict():
        pts = []                                   # (B, reserved_bytes) from SUCCESSFUL probes (over-budget is fine)
        b0 = base
        while b0 >= 1:
            _, pv = fits(b0)
            if pv is not None:
                pts.append((b0, pv)); break
            b0 //= 2                               # OOM -> cannot even measure; halve
        if not pts:
            return None
        b1 = max(1, pts[0][0] // 2)
        if b1 == pts[0][0]:
            return None                            # only one probeable point (base already 1)
        _, pv1 = fits(b1)
        if pv1 is None:
            return None
        pts.append((b1, pv1))
        (x0, y0), (x1, y1) = pts
        a = (y0 - y1) / (x0 - x1)
        if a <= 0:
            log(f"[autobatch] fit rejected: non-positive slope from {pts} — falling back to bisection")
            return None
        b_fit = y0 - a * x0
        log(f"[autobatch] fit: {a/1e9:.3f} GB/sample, intercept {b_fit/1e9:.2f}GB "
            f"(analytic persistent {_persistent_bytes()/1e9:.2f}GB) from B={x0},{x1}")
        for _ in range(3):                         # predict -> confirm -> re-fit with the new point
            pred = int((budget - b_fit) // a)
            pred = max(1, min(pred, cap))
            okp, pp = fits(pred)
            if okp:
                log(f"[autobatch] fit chose data.batch={pred} ({(pp or 0)/1e9:.1f}/{budget/1e9:.0f}GB reserved)")
                return pred
            if pp is None or pred <= 1:
                break
            a = (pp - y1) / max(1, pred - x1)      # re-fit against the newly measured over-budget point
            if a <= 0: break
            b_fit = pp - a * pred
            log(f"[autobatch] batch {pred} over budget ({pp/1e9:.1f}GB) — re-fit to {a/1e9:.3f} GB/sample")
        log("[autobatch] fit did not converge — falling back to bisection")
        return None

    _fb = _fit_predict()
    if _fb is not None:
        return done(confirm_compiled(_fb))
    # ---- fallback: the original bracket-and-bisect search (unchanged behaviour) ---------------------------
    ok, p = fits(base)
    if not ok:
        # base itself is over budget (or OOM'd while probing -> p is None). DON'T return base — that would launch a
        # run doomed to OOM at epoch 0 (this is exactly what small+F64+unet-decode hit on 2026-08-06). Step DOWN,
        # halving, until a batch fits; only then return it (compiled-confirmed).
        _pm = lambda v: "OOM while probing" if v is None else f"{v/1e9:.1f}/{budget/1e9:.0f}GB"
        log(f"[autobatch] base batch {base} over budget ({_pm(p)}) — searching below base")
        b = base // 2
        while b >= 1:
            okb, pb = fits(b)
            if okb:
                # BISECT UPWARD (fix 2026-08-18). This branch used to RETURN the first halving that fit, with no
                # upward search -- so a base of 16 that did not fit landed on 8 and never tried 9..15. Measured
                # cost: bsp32mse_long probed 82.2GB at batch 16 and 41.2GB at batch 8, so it ran at 41 of a 65GB
                # budget (32% of a 95.8GB card) at 61% GPU utilisation, dispatch-bound, for want of ~batch 12.
                # The base-FITS branch below always bisected properly; only this one did not. Granularity 1 here
                # because the interesting range (8..16) contains no multiple of 8 to bisect to.
                lo2, hi2 = b, b * 2                       # lo2 fits, hi2 does not
                while hi2 - lo2 > 1:
                    mid = (lo2 + hi2) // 2
                    okm, pm = fits(mid)
                    if okm: lo2, pb = mid, pm
                    else: hi2 = mid
                log(f"[autobatch] fits below base: data.batch={lo2} ({_pm(pb)} @ "
                    f"reserve {reserve/1e9:.0f}GB; bisected in [{b}, {b*2}))")
                return done(confirm_compiled(lo2))
            log(f"[autobatch] batch {b} over budget ({_pm(pb)}) — halving")
            b //= 2
        raise RuntimeError(f"[autobatch] even batch 1 exceeds the {budget/1e9:.0f}GB budget — this model+F+recon_frac "
                           f"does not fit; lower model.size / data.F / recon_frac (or lower data.autobatch_reserve_gb).")
    lo = hi = base
    while hi * 2 <= cap:                       # double to bracket [lo fits, hi over]
        ok, p = fits(hi * 2)
        if ok: lo = hi = hi * 2
        else: hi = hi * 2; break
    if lo == hi:                               # fit to the cap without going over
        log(f"[autobatch] fits to cap: data.batch={lo} (VRAM {total/1e9:.0f}GB, reserve {reserve/1e9:.0f}GB, cap {cap})")
        return done(confirm_compiled(lo))
    while hi - lo > 1:                          # bisect to EXACT resolution (was: multiples of 8, which on a
        mid = (lo + hi) // 2                    #   [8,16) bracket had no landing point at all and silently
        if mid <= lo or mid >= hi: break        #   returned the low end -- see the below-base branch comment
        ok, p = fits(mid)
        if ok: lo = mid
        else: hi = mid
    log(f"[autobatch] chose data.batch={lo}  (VRAM {total/1e9:.0f}GB, budget {budget/1e9:.0f}GB, "
        f"reserve {reserve/1e9:.0f}GB; probed rollout_train fwd+decode+bwd)")
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


def effective_action_dim(cfg) -> int:
    """`model.action_dim` AS THE MODEL SEES IT. `data.action_aggregate=concat` keeps all `subsample` raw
    actions of each kept step instead of folding them into one, so the action vector is subsample x wider.
    DERIVED, never hand-set: a hand-set width goes stale the moment subsample changes, and the failure is a
    shape error deep in the first batch rather than at config time."""
    a = int(cfg.model.get("action_dim", 2))
    if str(cfg.data.get("action_aggregate", "sum")) == "concat":
        a *= max(1, int(cfg.data.get("subsample", 1) or 1))
    return a


def normalizer(cfg) -> Normalizer:
    n = Normalizer.from_file(resolve_data_root(cfg)).subset_obs()   # subset via the process-wide set_obs_keep
    if str(cfg.data.get("action_aggregate", "sum")) == "concat":    # one step carries `subsample` raw actions
        n = n.tile_act(int(cfg.data.get("subsample", 1) or 1))
    return n


def image_head_cams(cfg) -> dict[str, str]:
    """{image modality name: camera stream} from the model config. THE one place head->camera is resolved.

    Every image modality names its own camera via `ModalitySpec.cam`; a SINGLE image head may leave it unset
    and fall back to `data.cam`, so single-camera configs are untouched. With more than one head an unset
    `cam` is an ERROR rather than a fallback: falling back would point every head at the same camera, and a
    head scored against another camera's frames is a plausible WRONG NUMBER rather than a crash.

    Exists so `window_loaders` and all eight evaluation load sites resolve it IDENTICALLY -- they used to
    each read `data.cam` directly, which is how the evals ended up feeding one camera to every head."""
    specs = [sp for sp in _modality_specs(cfg) if sp.kind == "image"]
    default_cam = str(cfg.data.get("cam", "fpv"))
    if len(specs) > 1:
        missing = [sp.name for sp in specs if not getattr(sp, "cam", None)]
        assert not missing, (f"{len(specs)} image modalities but {missing} have no `cam:` -- with more than "
                             f"one image head each must name its own camera, or they all read data.cam="
                             f"{default_cam!r} and the extra heads are scored against the wrong frames")
    out = {sp.name: str(getattr(sp, "cam", None) or default_cam) for sp in specs}
    if len(out) > 1:
        assert len(set(out.values())) == len(out), f"two image modalities share a camera: {out}"
    return out


def image_head_sizes(cfg) -> dict[str, int]:
    """{image modality name: img_size}. Companion to image_head_cams -- heads may differ in resolution."""
    return {sp.name: sp.img_size for sp in _modality_specs(cfg) if sp.kind == "image"}


def window_loaders(cfg, norm: Normalizer):
    """The ONE GPU-resident loader for every model. Loads the FPV frame store only when an image modality
    is present; proprio-only just loads (obs, act) — no frames touched."""
    P, F = cfg.data.P, cfg.data.F
    root = resolve_data_root(cfg)
    specs = _modality_specs(cfg)
    # EVERY image modality gets its own camera stream. This used to be `next(... kind == "image")`, which
    # took the FIRST image spec and silently ignored the rest: a second image modality was BUILT in the
    # model but never LOADED, so lit.py's `obs[name] = batch[name]` KeyError'd at step 0 -- nothing at
    # config time said anything was wrong. See design/two_camera_plan.md.
    head_cams = image_head_cams(cfg)          # {head: camera} -- the ONE resolution point
    head_sizes = image_head_sizes(cfg)
    repo = str(cfg.data.get("repo_id", "torus"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = {}
    for split, shuffle in (("train", True), ("val", False)):
        stride = int(cfg.data.get("window_stride", 1)) if split == "train" else 1   # subsample TRAIN windows only; val stays dense
        if head_cams:
            eps = load_split_episodes_mm(root, split, img_size=head_sizes, cam=head_cams, repo_id=repo)
            loaders[split] = MMWindowLoader(eps, P, F, norm, cfg.data.batch, shuffle, dev,
                                            image_head=list(head_cams), stride=stride)
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
        # 2026-08-11: act_enc and every VectorModality.enc became a FourierMLP wrapping the old Sequential as
        # `.net`, so `act_enc.0.weight` -> `act_enc.net.0.weight`. Migrate the OLD layout forward, otherwise a
        # pre-existing checkpoint loads with those encoders left at RANDOM INIT and says nothing (strict=False).
        for _pre in ("act_enc", "enc"):
            k = re.sub(rf"(^|\.)({_pre})\.(\d+)\.", rf"\1\2.net.\3.", k)
        clean[k] = v
    inc = model.load_state_dict(clean, strict=False)
    # strict=False is REQUIRED (buffers like lat_mean/lat_std are absent from older checkpoints), but silently
    # discarding IncompatibleKeys is how a partly-RANDOM model gets evaluated as if it were trained. Missing
    # PARAMETERS are fatal; missing buffers that have a defined default are reported and tolerated.
    _param_names = {n for n, _ in model.named_parameters()}
    _missing_params = [k for k in inc.missing_keys if k in _param_names]
    if inc.missing_keys or inc.unexpected_keys:
        print(f"[load_checkpoint] missing={len(inc.missing_keys)} unexpected={len(inc.unexpected_keys)}"
              + (f"\n  missing: {inc.missing_keys[:8]}" if inc.missing_keys else "")
              + (f"\n  unexpected: {inc.unexpected_keys[:8]}" if inc.unexpected_keys else ""))
    if _missing_params:
        raise RuntimeError(
            f"checkpoint is missing {len(_missing_params)} PARAMETER tensors, so those modules would stay at "
            f"random init and be evaluated as if trained: {_missing_params[:10]}. This is usually a layout "
            f"change between the checkpoint and the current code -- add a migration to load_checkpoint rather "
            f"than evaluating a partly-random model."
        )
    return model
