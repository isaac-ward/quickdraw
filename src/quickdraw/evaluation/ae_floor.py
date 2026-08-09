"""AE-floor Phase-1 gate helpers (issue #12): the raw-TAESD domain-shift probe + the num_tokens-floor note.

These measure the image encode->decode CEILING WITHOUT training the dynamics — the decisive go/no-go signals
before building the TAESD modality + adapters (Phase 2). `eval_ae_floor` (evaluation/routines.py) measures the
BESPOKE AE's floor on a checkpoint; the two functions here cover the questions that need no dynamics at all.
"""
from __future__ import annotations

import numpy as np
import torch


def _psnr_ssim(gt, rec):
    """gt/rec: (N,H,W,3) float [0,1] tensors -> (psnr_db, ssim, mse)."""
    from .openloop import _ssim
    gt, rec = gt.clamp(0, 1), rec.clamp(0, 1)
    mse = float(torch.mean((rec - gt) ** 2))
    psnr = -10.0 * np.log10(max(mse, 1e-12))
    ssim = float(max(0.0, min(1.0, _ssim(rec, gt))))
    return psnr, ssim, mse


def raw_taesd_floor(cfg, sizes=(128,), n_frames=32, device=None, log=print):
    """RAW pretrained-TAESD floor: load `diffusers.AutoencoderTiny(pretrained_name)` and round-trip N REAL val
    frames with NO training and NO adapter -> PSNR/SSIM. Tests DOMAIN SHIFT (TAESD trained on natural ~512^2 web
    images vs synthetic MuJoCo/ISS renders at 128) — the one Phase-1 signal that needs zero training (#12 §5).
    diffusers is imported directly (already importable in the container; a transitive dep — declare it if Phase 2
    lands, #12 §1). Runs on CPU by default (TAESD is tiny) to avoid contending with training GPUs.

    Pixel convention is auto-detected per size: AutoencoderTiny's expected input range varies by version, so we
    try both the [-1,1] (standard-VAE) and [0,1] round-trips and report the one with the higher PSNR (the correct
    scaling wins decisively) — so the reported floor is not confounded by a wrong normalization."""
    from diffusers import AutoencoderTiny

    from ..data.dataset import load_fpv_frames
    from ..training.setup import resolve_data_root
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    name = str(_af(cfg).get("pretrained_name", "madebyollin/taesd"))
    log(f"[raw_taesd] loading {name} on {dev} (no training, no adapter)...")
    ae = AutoencoderTiny.from_pretrained(name).to(dev).eval()
    root, cam = resolve_data_root(cfg), cfg.data.get("cam", "fpv")
    out = {}
    for sz in sizes:
        frames = load_fpv_frames(root, "val", size=sz, cam=cam, max_frames=n_frames, cache=False)   # (n,sz,sz,3) uint8
        gt = torch.from_numpy(np.asarray(frames[:n_frames])).float().div(255.0).to(dev)              # (n,sz,sz,3) [0,1]
        x = gt.permute(0, 3, 1, 2)                                                                    # (n,3,sz,sz)
        best = None
        for conv, inp, back in (("[-1,1]", x * 2 - 1, lambda d: (d + 1) / 2), ("[0,1]", x, lambda d: d)):
            with torch.no_grad():
                lat = ae.encode(inp).latents
                dec = ae.decode(lat).sample
            rec = back(dec).permute(0, 2, 3, 1)                                                       # (n,sz,sz,3)
            psnr, ssim, mse = _psnr_ssim(gt, rec)
            if best is None or psnr > best["psnr"]:
                best = {"psnr": psnr, "ssim": ssim, "mse": mse, "convention": conv,
                        "latent_shape": tuple(lat.shape[1:]), "latent_floats": int(np.prod(lat.shape[1:]))}
        out[sz] = best
        log(f"[raw_taesd] {sz}x{sz}: PSNR={best['psnr']:.2f}dB SSIM={best['ssim']:.3f} MSE={best['mse']:.5f} "
            f"| pixel-conv {best['convention']} | TAESD latent {best['latent_shape']} = {best['latent_floats']} floats "
            f"(vs bespoke num_tokens*d budget)")
    return out


def num_tokens_floor_note(log=print):
    """The num_tokens ∈ {8,16,32} bespoke-AE floor sweep (#12 §5) — INFRA + honest caveat, NOT a fabricated number.

    A meaningful bespoke-AE floor requires a TRAINED AE: an untrained encoder/decoder reconstructs garbage, so
    running eval_ae_floor on a fresh model would report a floor that says nothing about the tokenizer's capacity.
    So this is intentionally a runner over TRAINED checkpoints, not a one-shot on an untrained model:

        # option A — sweep eval_ae_floor over existing trained checkpoints at each width:
        for nt in 8 16 32:
          python -m quickdraw.eval_ae_floor checkpoint=<run_trained_at_nt> 'model.modalities.1.num_tokens='$nt

        # option B — short AE-only runs per width (freeze dynamics / few epochs), then eval_ae_floor on each.

    The single measurement it yields — floor(num_tokens) — decides whether the ceiling is the BOTTLENECK (floor
    rises with num_tokens) or the TOKENIZER (floor flat), i.e. whether raising num_tokens or swapping in a
    pretrained AE is the lever. It needs no dynamics, but it DOES need a trained AE, hence checkpoints."""
    log(num_tokens_floor_note.__doc__)


def _af(cfg):
    """The optional `ae_floor` config block (added via `+ae_floor.<k>=<v>` overrides); {} if absent."""
    try:
        return cfg.get("ae_floor", {}) or {}
    except Exception:
        return {}


def assert_identity_floor(cfg, model, log=print, n_frames=8, tol_db=0.1):
    """EPOCH-0 GATE (#12 §4). For every pretrained-AE image trunk, round-trip REAL val frames through
    encode->decode and compare against the SAME (frozen) AE used raw, with no adapter in the path.

    When the adapter mode is EXACT/PADDED the base path is a parameter-free index rearrangement and the
    residual is zero-initialised, so the round-trip is the IDENTITY at step 0 and the two PSNRs must agree to
    float noise -> ASSERT. When the mode is PROJECTED there is a learned per->d map with no such guarantee, so
    the delta is REPORTED and left to the round-trip loss to close. This turns "the tokenizer is faithful" from
    a hope into a precondition, which is what the 10.5 dB learned-Perceiver collapse cost us the first time."""
    import numpy as _np
    import torch as _t

    from ..data.dataset import load_fpv_frames
    from ..training.setup import resolve_data_root
    m = getattr(model, "_orig_mod", model)
    mods = [(n, md) for n, md in getattr(m, "modalities", {}).items() if getattr(md, "adapter_info", None)]
    if not mods:
        return {}
    root, cam = resolve_data_root(cfg), cfg.data.get("cam", "fpv")
    out = {}
    for name, mod in mods:
        info = mod.adapter_info
        sz = mod.ae.cfg.img_size
        dev = next(mod.parameters()).device
        frames = load_fpv_frames(root, "val", size=sz, cam=cam, max_frames=n_frames, cache=False)
        gt = _t.from_numpy(_np.asarray(frames)).float().div(255.0).to(dev)
        was = mod.training
        mod.eval()
        with _t.no_grad():
            rec = mod.decode(mod.encode(gt)).clamp(0, 1)                       # through the ADAPTERS
            x = gt.permute(0, 3, 1, 2) * 2 - 1                                  # raw AE, no adapter
            raw = ((mod.taesd.decode(mod.taesd.encode(x).latents).sample + 1) / 2).permute(0, 2, 3, 1).clamp(0, 1)
        if was:
            mod.train()
        p_ad = float(-10.0 * _np.log10(max(float(((rec - gt) ** 2).mean()), 1e-12)))
        p_raw = float(-10.0 * _np.log10(max(float(((raw - gt) ** 2).mean()), 1e-12)))
        delta = p_ad - p_raw
        out[name] = {"adapter_db": p_ad, "raw_db": p_raw, "delta_db": delta, "mode": info["mode"]}
        if info["identity_at_init"]:
            ok = abs(delta) < tol_db
            log(f"[ae_floor @ep0] {name}: adapter {p_ad:.2f} dB | raw AE {p_raw:.2f} dB | delta {delta:+.3f} dB "
                f"| {info['mode']} -> {'OK' if ok else 'MISMATCH'}")
            if not ok:
                raise AssertionError(
                    f"pretrained-AE adapter '{name}' is mode {info['mode']}, which GUARANTEES an identity "
                    f"round-trip at init, but the epoch-0 floor is {p_ad:.2f} dB vs the raw AE's {p_raw:.2f} dB "
                    f"(delta {delta:+.3f} dB > {tol_db}). The pad/strip indexing or the zero-init residual is "
                    f"wrong — fix that rather than training through it.")
        else:
            log(f"[ae_floor @ep0] {name}: adapter {p_ad:.2f} dB | raw AE {p_raw:.2f} dB | delta {delta:+.3f} dB "
                f"| {info['mode']} -> no identity guarantee, round-trip must be LEARNED (watch this climb)")
    return out
